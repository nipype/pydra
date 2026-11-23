"""Process groups for process-owning providers (RFC 0005 section 3.1).

A provider whose ``load`` starts a child (a model server, a worker in another interpreter)
returns a handle over a :class:`ProcessGroup`. ``close`` stops the whole group and every
descendant, waits until they are gone, and waits until the device memory they held is free.
Only a provider that starts a process imports this module.

On POSIX the group's leader is a shim that starts the command in its own session and stops the
tree when the parent process dies (its lifeline is a pipe the parent holds); on Linux it is also a
child subreaper, so a server its launcher daemonised is stopped with the rest. On other POSIX
systems a daemonised server whose parent has exited is not stopped. On Windows the child gets a
new process group and is stopped with ``taskkill /T /F``, with no lifeline.
"""

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import typing as ty
from pathlib import Path

from pydra.engine._procshim import _descendants, _process_table, _Row
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
_LOGGED_NO_ACCOUNTING = False
_SHIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_procshim.py")


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

    def __init__(self, proc: subprocess.Popen, lifeline: int | None = None):
        self.proc = proc
        self._lifeline = lifeline
        self._known: dict[int, ty.Optional[str]] = {}  # pid -> start identity
        self._stopped = False

    @classmethod
    def start(
        cls,
        argv: ty.Sequence[str],
        *,
        env: ty.Mapping[str, str] | None = None,
        cwd: str | os.PathLike | None = None,
        **popen_kwargs: ty.Any,
    ) -> "ProcessGroup":
        """Start ``argv`` as the leader of a new process group (on POSIX, under the
        lifeline shim)"""
        if sys.platform == "win32":
            return cls(
                subprocess.Popen(
                    list(argv),
                    env=None if env is None else dict(env),
                    cwd=cwd,
                    **popen_group_kwargs(),
                    **popen_kwargs,
                )
            )
        read_fd, write_fd = os.pipe()
        # the command's argv, env and cwd go to the shim in an anonymous file; the shim
        # itself runs isolated, in the parent's environment and a neutral directory
        cfg = tempfile.TemporaryFile("w+")
        json.dump(
            {
                "argv": [str(a) for a in argv],
                "env": None if env is None else dict(env),
                "cwd": None if cwd is None else os.fspath(cwd),
            },
            cfg,
        )
        cfg.flush()
        try:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    _SHIM,
                    str(read_fd),
                    str(cfg.fileno()),
                ],
                pass_fds=(read_fd, cfg.fileno()),
                cwd=os.sep,
                **popen_group_kwargs(),
                **popen_kwargs,
            )
        except BaseException:
            os.close(write_fd)
            raise
        finally:
            os.close(read_fd)
            cfg.close()
        return cls(proc, lifeline=write_fd)

    @property
    def pid(self) -> int:
        return self.proc.pid

    # -- liveness, without reaping -------------------------------------------------

    def leader_running(self) -> bool:
        """Whether the leader has not exited. It is not reaped by looking, so its pid
        (and the group id) stay reserved until :meth:`stop`"""
        if self._stopped or self.proc.returncode is not None:
            return False
        if hasattr(os, "waitid") and sys.platform != "win32":
            try:
                done = os.waitid(
                    os.P_PID, self.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
                )
            except ChildProcessError:
                return False
            return done is None
        if sys.platform == "win32":
            return self.proc.poll() is None
        row = _process_table().get(self.pid)
        return row is not None and row[2]

    def _refresh(self, table: dict[int, _Row] | None = None) -> dict[int, _Row]:
        """Add the live descendants and group members to what is known"""
        table = _process_table() if table is None else table
        if sys.platform == "win32":
            return table
        for pid in [p for p in self._known if not self._same(p, table)]:
            del self._known[pid]  # exited, or the pid now belongs to another process
        mine = _descendants(table, self.pid)
        mine |= {p for p, row in table.items() if row[1] == self.pid}
        for pid in mine:
            row = table.get(pid)
            if row is not None and row[2]:
                self._known.setdefault(pid, row[3])
        return table

    def _same(self, pid: int, table: dict[int, _Row]) -> bool:
        """``pid`` is running and is the process first seen under that pid"""
        row = table.get(pid)
        if row is None or not row[2]:
            return False
        start = self._known.get(pid)
        return start is None or row[3] is None or row[3] == start

    def _live_known(self, table: dict[int, _Row]) -> list[int]:
        return [pid for pid in self._known if self._same(pid, table)]

    def members(self) -> list[int]:
        """Pids of the live (not zombie) members of the group and its descendants"""
        if self._stopped:
            return []
        if sys.platform == "win32":
            return [self.pid] if self.leader_running() else []
        table = self._refresh()
        return sorted(self._live_known(table))

    def known_pids(self) -> list[int]:
        """Every pid seen in the group or below it, live or not"""
        return sorted(self._known)

    def _signal_all(self, sig: int, table: dict[int, _Row]) -> None:
        try:
            if self._group_has_members(table):
                os.killpg(self.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        for pid in self._live_known(table):
            if table[pid][1] == self.pid:
                continue  # reached through the group
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass

    def _group_has_members(self, table: dict[int, _Row]) -> bool:
        return any(row[1] == self.pid and row[2] for row in table.values())

    def _wait_all_gone(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            table = self._refresh()
            if not self._live_known(table) and not self._group_has_members(table):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(_POLL)

    def stop(self, term_timeout: float = 10.0, kill_timeout: float = 10.0) -> None:
        """SIGTERM the group and every descendant, wait up to ``term_timeout``, then
        SIGKILL, and reap the leader last. Returns when every member is gone (or, after
        ``kill_timeout`` past the kill, logs that one is unkillable). A second call does
        nothing"""
        if self._stopped:
            return
        try:
            if sys.platform == "win32":
                self._stop_windows(term_timeout, kill_timeout)
                return
            table = self._refresh()
            self._signal_all(signal.SIGTERM, table)
            if not self._wait_all_gone(term_timeout):
                self._signal_all(signal.SIGKILL, self._refresh())
                if not self._wait_all_gone(kill_timeout):
                    logger.error("Process group %s survived SIGKILL", self.pid)
            try:
                self.proc.wait(timeout=1)  # the leader is a zombie or dead: reap it
            except subprocess.TimeoutExpired:
                logger.error("Leader %s of a process group did not exit", self.pid)
        finally:
            self._stopped = True
            if self._lifeline is not None:
                try:
                    os.close(self._lifeline)
                except OSError:
                    pass
                self._lifeline = None

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
    """The ``oom_kill`` count of the process's cgroup, taken as a baseline after each
    provider start so a later kill is charged to the call that ran it.

    The counter is read from cgroup v2 ``memory.events``, or on cgroup v1 from
    ``memory.oom_control`` (kernel 4.13 and later). ``path`` is ``None`` (no accounting;
    :meth:`take` returns ``None``) when neither file is present: a file is read only
    when it exists.
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
        """The baseline over the calling process's own cgroup"""
        global _LOGGED_NO_ACCOUNTING
        try:
            lines = Path(proc_cgroup).read_text().splitlines()
        except OSError:
            lines = []
        for line in lines:
            hierarchy, _, rest = line.partition(":")
            controllers, _, path = rest.partition(":")
            relative = path.lstrip("/")
            if hierarchy == "0":
                candidate = Path(root) / relative / "memory.events"
            elif "memory" in controllers.split(","):
                candidate = Path(root) / "memory" / relative / "memory.oom_control"
            else:
                continue
            if candidate.is_file():
                return cls(candidate)
        if not _LOGGED_NO_ACCOUNTING:
            _LOGGED_NO_ACCOUNTING = True
            logger.debug(
                "No readable cgroup oom_kill counter: OOM accounting for process-owning "
                "providers is off"
            )
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
        readable counter"""
        current = self._read()
        if current is None:
            return None
        killed, self._base = max(current - self._base, 0), current
        return killed


_BASELINE: CgroupOomBaseline | None = None


def process_baseline() -> CgroupOomBaseline:
    """The one OOM baseline of this process (the worker's), shared by every provider"""
    global _BASELINE
    if _BASELINE is None:
        _BASELINE = CgroupOomBaseline.discover()
    return _BASELINE


class ProcessHandle:
    """What a :class:`ProcessProvider` ``load`` returns: the group and the client built
    over it. The OOM baseline is the process's, not the handle's"""

    def __init__(self, group: ProcessGroup, client: ty.Any) -> None:
        self.group = group
        self.client = client

    @property
    def pid(self) -> int:
        return self.group.pid

    def take_oom(self) -> int | None:
        """OOM kills charged since the baseline or the last call, to whichever call asks
        first (see :class:`CgroupOomBaseline`)"""
        return process_baseline().take()


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
        """The child's environment; ``None`` inherits. A CUDA device index is relative
        to this process's own ``CUDA_VISIBLE_DEVICES`` when that is set"""
        if device.kind == "cuda" and device.index is not None:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if visible.strip():
                entries = [e.strip() for e in visible.split(",")]
                if device.index >= len(entries):
                    raise ValueError(
                        f"Device index {device.index} is outside "
                        f"CUDA_VISIBLE_DEVICES={visible!r}"
                    )
                value = entries[device.index]
            else:
                value = str(device.index)
            return {**os.environ, "CUDA_VISIBLE_DEVICES": value}
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
        process_baseline().reset()  # after the start, so a load-time kill is not charged
        return ProcessHandle(group, client)

    def close(self, obj: object) -> None:
        """Stop the group, wait for every member, then for its device memory to be
        free. Errors are logged, not raised"""
        handle = ty.cast(ProcessHandle, obj)
        try:
            handle.group.members()  # snapshot the descendants before signalling
            pids = handle.group.known_pids()
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
