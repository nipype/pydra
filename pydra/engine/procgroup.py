"""Process groups for process-owning providers (RFC 0005 section 3.1, W2b).

A provider whose ``load`` starts a child (a model server, a worker in another
interpreter) returns a handle over a :class:`ProcessGroup`. ``close`` stops the whole
group and waits until every member is gone, and until the device memory the group held
is free, so a restart does not race the dying process.

Only a provider that starts a process imports this module; a plain run does not.

On POSIX the child starts in its own session, so its process group id is its pid and a
signal to the group reaches every descendant that did not leave it. On Windows the child
is created with ``CREATE_NEW_PROCESS_GROUP`` and is stopped with ``taskkill /T /F``,
which walks the process tree.
"""

import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import typing as ty
from pathlib import Path

from pydra.engine.resident import Device, Footprint

__all__ = [
    "CgroupOomBaseline",
    "ProcessGroup",
    "ProcessHandle",
    "ProcessProvider",
    "gpu_memory_by_pid",
    "wait_device_memory_free",
]

logger = logging.getLogger("pydra.resident")

_POLL = 0.02
_CREATE_NEW_PROCESS_GROUP = 0x00000200


def popen_group_kwargs(platform: str = sys.platform) -> dict[str, ty.Any]:
    """The ``Popen`` keywords that put a child in a group of its own"""
    if platform == "win32":
        return {
            "creationflags": getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", _CREATE_NEW_PROCESS_GROUP
            )
        }
    return {"start_new_session": True}


def windows_kill_tree_command(pid: int) -> list[str]:
    """The command that ends ``pid`` and its descendants on Windows"""
    return ["taskkill", "/T", "/F", "/PID", str(pid)]


class ProcessGroup:
    """A child process and the group it leads"""

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc

    @classmethod
    def start(
        cls,
        argv: ty.Sequence[str],
        *,
        env: ty.Mapping[str, str] | None = None,
        cwd: str | os.PathLike | None = None,
        **popen_kwargs: ty.Any,
    ) -> "ProcessGroup":
        """Start ``argv`` as the leader of a new process group"""
        proc = subprocess.Popen(
            list(argv),
            env=None if env is None else dict(env),
            cwd=cwd,
            **popen_group_kwargs(),
            **popen_kwargs,
        )
        return cls(proc)

    @property
    def pid(self) -> int:
        return self.proc.pid

    def members(self) -> list[int]:
        """Pids of the live members of the group (the leader's pid alone on Windows,
        where the tree is not enumerated)"""
        alive = self.proc.poll() is None
        if sys.platform == "win32" or not _group_exists(self.pid):
            return [self.pid] if alive else []
        try:
            out = subprocess.run(
                ["ps", "-axo", "pid=,pgid="],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return [self.pid] if self.proc.poll() is None else []
        pids = []
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == str(self.pid):
                pids.append(int(parts[0]))
        return pids

    def _gone(self) -> bool:
        """The leader is reaped and no member of its group remains"""
        if self.proc.poll() is None:
            return False
        return sys.platform == "win32" or not _group_exists(self.pid)

    def _signal(self, sig: int) -> None:
        try:
            os.killpg(self.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _wait_gone(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            if self._gone():
                return True
            if time.monotonic() >= deadline:
                return False
            try:
                self.proc.wait(timeout=_POLL)
            except subprocess.TimeoutExpired:
                time.sleep(_POLL)

    def stop(self, term_timeout: float = 10.0, kill_timeout: float = 10.0) -> None:
        """SIGTERM the group, wait up to ``term_timeout``, then SIGKILL it; always reap
        the leader. Returns when every member is gone (or, after ``kill_timeout`` past
        the kill, logs that one is unkillable). Safe to call again"""
        if sys.platform == "win32":
            self._stop_windows(term_timeout, kill_timeout)
            return
        self._signal(signal.SIGTERM)
        if not self._wait_gone(term_timeout):
            self._signal(signal.SIGKILL)
            if not self._wait_gone(kill_timeout):
                logger.error("Process group %s survived SIGKILL", self.pid)
        self.proc.poll()

    def _stop_windows(self, term_timeout: float, kill_timeout: float) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                self.proc.wait(timeout=term_timeout)
            except (subprocess.TimeoutExpired, OSError):
                pass
        if self.proc.poll() is None:
            subprocess.run(
                windows_kill_tree_command(self.pid),
                capture_output=True,
                timeout=kill_timeout + 10,
            )
        try:
            self.proc.wait(timeout=kill_timeout)
        except subprocess.TimeoutExpired:
            logger.error("Process %s survived taskkill", self.pid)


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def gpu_memory_by_pid(
    pids: ty.Iterable[int] | None = None, timeout: float = 20.0
) -> dict[int, int] | None:
    """Bytes of device memory held by each compute process, from ``nvidia-smi
    --query-compute-apps``; ``None`` when ``nvidia-smi`` is absent or fails (there is
    then no GPU accounting), otherwise restricted to ``pids`` when given"""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [
                exe,
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    wanted = None if pids is None else set(pids)
    held: dict[int, int] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid, mib = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if wanted is None or pid in wanted:
            held[pid] = held.get(pid, 0) + mib * 1024 * 1024
    return held


def wait_device_memory_free(pids: ty.Iterable[int], timeout: float) -> bool:
    """Wait until none of ``pids`` is listed by ``nvidia-smi``. True when free, or when
    there is no GPU accounting; False on timeout"""
    pids = list(pids)
    deadline = time.monotonic() + timeout
    while True:
        held = gpu_memory_by_pid(pids)
        if not held:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


class CgroupOomBaseline:
    """The ``oom_kill`` count of a cgroup v2 ``memory.events`` file, taken as a baseline
    after each provider start so a later kill is charged to the call that ran it.

    ``path`` is ``None`` (no accounting; :meth:`take` returns ``None``) when the file is
    not present: the file is read only when it exists.
    """

    def __init__(self, path: str | os.PathLike | None):
        self.path = Path(path) if path is not None else None
        self._base = 0

    @classmethod
    def discover(
        cls,
        proc_cgroup: str | os.PathLike = "/proc/self/cgroup",
        root: str | os.PathLike = "/sys/fs/cgroup",
    ) -> "CgroupOomBaseline":
        """The baseline over the calling process's own cgroup, if it has a v2
        ``memory.events``"""
        try:
            for line in Path(proc_cgroup).read_text().splitlines():
                hierarchy, _, rest = line.partition(":")
                if hierarchy != "0":
                    continue
                relative = rest.partition(":")[2].lstrip("/")
                events = Path(root) / relative / "memory.events"
                if events.is_file():
                    return cls(events)
        except OSError:
            pass
        return cls(None)

    def _read(self) -> int | None:
        if self.path is None:
            return None
        try:
            for line in self.path.read_text().splitlines():
                key, _, value = line.partition(" ")
                if key == "oom_kill":
                    return int(value)
        except (OSError, ValueError):
            pass
        return None

    def reset(self) -> None:
        """Take the baseline now (after a provider start)"""
        current = self._read()
        self._base = current if current is not None else 0

    def take(self) -> int | None:
        """OOM kills since the baseline, which moves up to now; ``None`` without a
        readable ``memory.events``"""
        current = self._read()
        if current is None:
            return None
        killed, self._base = max(current - self._base, 0), current
        return killed


class ProcessHandle:
    """What a :class:`ProcessProvider` ``load`` returns: the group, the client built
    over it, and the cgroup OOM baseline taken after the start"""

    def __init__(
        self, group: ProcessGroup, client: ty.Any, oom: CgroupOomBaseline
    ) -> None:
        self.group = group
        self.client = client
        self.oom = oom

    @property
    def pid(self) -> int:
        return self.group.pid

    def take_oom(self) -> int | None:
        """OOM kills charged since the last call (see :class:`CgroupOomBaseline`)"""
        return self.oom.take()


class ProcessProvider:
    """Base for a provider whose state is a child process.

    Subclasses set ``name`` and implement :meth:`argv` (and optionally :meth:`env`,
    :meth:`connect`, :meth:`footprint`, :meth:`version`).
    """

    name: ty.ClassVar[str]

    def __init__(
        self,
        term_timeout: float = 10.0,
        kill_timeout: float = 10.0,
        gpu_settle_timeout: float = 30.0,
    ):
        self.term_timeout = term_timeout
        self.kill_timeout = kill_timeout
        self.gpu_settle_timeout = gpu_settle_timeout

    def argv(self, spec: ty.Any, device: Device) -> list[str]:
        raise NotImplementedError

    def env(self, spec: ty.Any, device: Device) -> dict[str, str] | None:
        """The child's environment; ``None`` inherits"""
        if device.kind == "cuda" and device.index is not None:
            return {**os.environ, "CUDA_VISIBLE_DEVICES": str(device.index)}
        return None

    def connect(self, group: ProcessGroup, spec: ty.Any) -> ty.Any:
        """The client of the started process (default: none)"""
        return None

    def load(self, spec: ty.Any, device: Device) -> ProcessHandle:
        group = ProcessGroup.start(self.argv(spec, device), env=self.env(spec, device))
        try:
            client = self.connect(group, spec)
        except BaseException:
            group.stop(self.term_timeout, self.kill_timeout)
            raise
        oom = CgroupOomBaseline.discover()
        oom.reset()
        return ProcessHandle(group, client, oom)

    def close(self, obj: object) -> None:
        """Stop the group, wait for every member, then for its device memory to be
        free. Errors are logged, not raised"""
        handle = ty.cast(ProcessHandle, obj)
        pids = handle.group.members() or [handle.pid]
        try:
            handle.group.stop(self.term_timeout, self.kill_timeout)
            if not wait_device_memory_free(pids, self.gpu_settle_timeout):
                logger.error(
                    "Device memory of %s still held after %ss",
                    self.name,
                    self.gpu_settle_timeout,
                )
        except Exception as exc:
            logger.error("Stopping process group %s failed: %r", handle.pid, exc)

    def footprint(self, spec: ty.Any) -> Footprint | None:
        return None

    def measure(self, obj: object) -> Footprint | None:
        """Device bytes held by the group's processes, by pid (``None`` without
        ``nvidia-smi``)"""
        handle = ty.cast(ProcessHandle, obj)
        pids = handle.group.members() or [handle.pid]
        held = gpu_memory_by_pid(pids)
        if held is None:
            return None
        return Footprint(device=sum(held.values()))

    def version(self) -> str:
        return type(self).__qualname__
